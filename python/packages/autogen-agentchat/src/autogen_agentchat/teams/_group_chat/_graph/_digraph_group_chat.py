import asyncio
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, List, Literal, Mapping, Sequence, Set, Union

from autogen_core import AgentRuntime, CancellationToken, Component, ComponentModel
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import ChatAgent, OrTerminationCondition, Response, TerminationCondition
from autogen_agentchat.conditions import StopMessageTermination
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ChatMessage,
    MessageFactory,
    StopMessage,
    TextMessage,
)
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams import BaseGroupChat, GraphFlowGroupChatRequestPublish, GraphFlowGroupChatAgentResponse

from ..._group_chat._base_group_chat_manager import BaseGroupChatManager
from ..._group_chat._events import GroupChatTermination

_DIGRAPH_STOP_AGENT_NAME = "DiGraphStopAgent"
_DIGRAPH_STOP_AGENT_MESSAGE = "Digraph execution is complete"


class DiGraphEdge(BaseModel):
    """Represents a directed edge in a :class:`DiGraph`, with an optional execution condition.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    .. warning::

        If the condition is a callable, it will not be serialized in the model.

    """

    target: str  # Target node name
    condition: Union[str, Callable[[BaseChatMessage], bool], None] = Field(default=None)
    """(Experimental) Condition to execute this edge.
    If None, the edge is unconditional.
    If a string, the edge is conditional on the presence of that string in the last agent chat message.
    If a callable, the edge is conditional on the callable returning True when given the last message.
    """

    # Using Field to exclude the condition in serialization if it's a callable
    condition_function: Callable[[BaseChatMessage], bool] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_condition(self) -> "DiGraphEdge":
        # Store callable in a separate field and set condition to None for serialization
        if callable(self.condition):
            self.condition_function = self.condition
            # For serialization purposes, we'll set the condition to None
            # when storing as a pydantic model/dict
            object.__setattr__(self, "condition", None)
        return self

    def check_condition(self, message: BaseChatMessage) -> bool:
        """Check if the edge condition is satisfied for the given message.

        Args:
            message: The message to check the condition against.

        Returns:
            True if condition is satisfied (None condition always returns True),
            False otherwise.
        """
        if self.condition_function is not None:
            return self.condition_function(message)
        elif isinstance(self.condition, str):
            # If it's a string, check if the string is in the message content
            return self.condition in message.to_model_text()
        return True  # None condition is always satisfied


class DiGraphNode(BaseModel):
    """Represents a node (agent) in a :class:`DiGraph`, with its outgoing edges and activation type.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    """

    name: str  # Agent's name
    edges: List[DiGraphEdge] = []  # Outgoing edges
    activation: Literal["all", "any"] = "all"


class DiGraph(BaseModel):
    """Defines a directed graph structure with nodes and edges.
    :class:`GraphFlow` uses this to determine execution order and conditions.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    """

    nodes: Dict[str, DiGraphNode]  # Node name → DiGraphNode mapping
    default_start_node: str | None = None  # Default start node name
    _has_cycles: bool | None = None  # Cyclic graph flag

    def get_parents(self) -> Dict[str, List[str]]:
        """Compute a mapping of each node to its parent nodes."""
        parents: Dict[str, List[str]] = {node: [] for node in self.nodes}
        for node in self.nodes.values():
            for edge in node.edges:
                if edge.target != node.name:
                    parents[edge.target].append(node.name)
        return parents

    def get_start_nodes(self) -> Set[str]:
        """Return the nodes that have no incoming edges (entry points)."""
        if self.default_start_node:
            return {self.default_start_node}

        parents = self.get_parents()
        return set([node_name for node_name, parent_list in parents.items() if not parent_list])

    def get_leaf_nodes(self) -> Set[str]:
        """Return nodes that have no outgoing edges (final output nodes)."""
        return set([name for name, node in self.nodes.items() if not node.edges])

    def has_cycles_with_exit(self) -> bool:
        """
        Check if the graph has any cycles and validate that each cycle has at least one conditional edge.

        Returns:
            bool: True if there is at least one cycle and all cycles have an exit condition.
                False if there are no cycles.

        Raises:
            ValueError: If there is a cycle without any conditional edge.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            for edge in self.nodes[node_name].edges:
                target = edge.target
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    # Found a cycle → extract the cycle
                    cycle_start_index = path.index(target)
                    cycle_nodes = path[cycle_start_index:]
                    cycle_edges: List[DiGraphEdge] = []
                    for n in cycle_nodes:
                        cycle_edges.extend(self.nodes[n].edges)
                    if all(edge.condition is None and edge.condition_function is None for edge in cycle_edges):
                        raise ValueError(
                            f"Cycle detected without exit condition: {' -> '.join(cycle_nodes + cycle_nodes[:1])}"
                        )
                    return True  # Found cycle, but it has an exit condition

            rec_stack.remove(node_name)
            path.pop()
            return False

        has_cycle = False
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    has_cycle = True

        return has_cycle

    def get_has_cycles(self) -> bool:
        """Indicates if the graph has at least one cycle (with valid exit conditions)."""
        if self._has_cycles is None:
            self._has_cycles = self.has_cycles_with_exit()

        return self._has_cycles

    def graph_validate(self) -> None:
        """Validate graph structure and execution rules."""
        if not self.nodes:
            raise ValueError("Graph has no nodes.")

        if not self.get_start_nodes():
            raise ValueError("Graph must have at least one start node")

        if not self.get_leaf_nodes():
            raise ValueError("Graph must have at least one leaf node")

        # Outgoing edge condition validation (per node)
        for node in self.nodes.values():
            # Check that if a node has an outgoing conditional edge, then all outgoing edges are conditional
            has_condition = any(
                edge.condition is not None or edge.condition_function is not None for edge in node.edges
            )
            has_unconditioned = any(edge.condition is None and edge.condition_function is None for edge in node.edges)
            if has_condition and has_unconditioned:
                raise ValueError(f"Node '{node.name}' has a mix of conditional and unconditional edges.")

        self._has_cycles = self.has_cycles_with_exit()

class DiGraphSCC(DiGraph):
    """Represents Strongly Connected Components in a directed graph
    
    Uses Tarjan's algorithm to find all strongly connected components and assigns unique identifiers to each SCC.
    Supports dependency management between SCCs and execution order calculation.
    """
    default_start_scc: List[str] | None = None
    
    class _TarjanDataStruct:
        """Data structure used by Tarjan's algorithm"""
        def __init__(self):
            self.time: int = 0
            self.disc: Dict[str, int] = {}  # Node discovery time
            self.low: Dict[str, int] = {}   # Minimum discovery time reachable by node
            self.on_stack: Set[str] = set() # Nodes currently on stack
            self.stack: List[str] = []      # DFS stack
            self.sccs: List[List[str]] = [] # Store found SCCs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use Tarjan's algorithm to find all SCCs
        self._sccs = self.find_sccs_tarjan()
        # Create unique identifiers for each SCC
        self._scc_counter = 0
        self._scc_nodes = {}  # scc_id -> nodes
        self._scc_adj = {}    # scc_id -> [neighbor_scc_ids]
        self._scc_edges :Dict[str, Dict[str, List[DiGraphEdge]]] = {}  # scc_id -> [source_node -> list[edges]]
        self._scc_in_degree = {}  # scc_id -> in_degree
        self._scc_nodes_in_degree :Dict[str, Dict[str, int]] = {}  # scc_id -> [node -> in_degree]
        self._scc_nodes_activation :Dict[str, Dict[str, Literal["any", "all"]]] = {}  # scc_id -> [node -> activation]
        
        # Initialize SCC data
        for scc in self._sccs:
            scc_id = f"scc_{self._scc_counter}"
            self._scc_counter += 1
            self._scc_nodes[scc_id] = scc
            self._scc_adj[scc_id] = []
            self._scc_edges[scc_id] = {}
            self._scc_in_degree[scc_id] = 0
            self._scc_nodes_in_degree[scc_id] = {node: 0 for node in scc}
            self._scc_nodes_activation[scc_id] = {}
        # Calculate dependencies between SCCs
        self._compute_scc_dependencies()
        # Calculate edges within each SCC
        self._compute_scc_edges()

    def _compute_scc_dependencies(self) -> None:
        """Calculate dependencies between SCCs"""
        # Create mapping from nodes to SCCs
        node_to_scc = {}
        for scc_id, nodes in self._scc_nodes.items():
            for node in nodes:
                node_to_scc[node] = scc_id

        # Calculate edges and in-degrees between SCCs
        for u, node in self.nodes.items():
            for edge in node.edges:
                v = edge.target
                scc_u = node_to_scc[u]
                scc_v = node_to_scc[v]
                
                # If edge connects two different SCCs, add dependency
                if scc_u != scc_v:
                    self._scc_adj[scc_u].append(scc_v)
                    self._scc_in_degree[scc_v] += 1
                    #compute edge
                if u not in self._scc_edges[scc_u]:
                    self._scc_edges[scc_u][u] = []
                self._scc_edges[scc_u][u].append(edge)

    def _compute_scc_edges(self) -> None:
        """Calculate edges within each SCC"""
        for scc_id, nodes in self._scc_nodes.items():
            scc_nodes = set(nodes)
            internal_edges = []
            
            for u in nodes:
                for edge in self.nodes[u].edges:
                    if edge.target in scc_nodes:
                        internal_edges.append(edge)
                    if edge.target not in self._scc_nodes_in_degree[scc_id]:
                        self._scc_nodes_in_degree[scc_id][edge.target] = 0
                    self._scc_nodes_in_degree[scc_id][edge.target] += 1
                    self._scc_nodes_activation[scc_id][edge.target] = "all"
                self._scc_edges[scc_id][u] = internal_edges

    def get_scc_nodes_in_degree(self) -> Dict[str, Dict[str, int]]:
        return self._scc_nodes_in_degree

    def get_scc_nodes_activation(self) -> Dict[str, Dict[str, Literal["any", "all"]]]:
        return self._scc_nodes_activation

    def _tarjan_dfs(self, u: str, tarjan_data: _TarjanDataStruct) -> None:
        """DFS implementation of Tarjan's algorithm"""
        tarjan_data.disc[u] = tarjan_data.low[u] = tarjan_data.time
        tarjan_data.time += 1
        tarjan_data.stack.append(u)
        tarjan_data.on_stack.add(u)
        
        adjs = self.nodes.get(u)
        if adjs:
            for edge in adjs.edges:
                v = edge.target
                if v not in tarjan_data.disc:
                    self._tarjan_dfs(v, tarjan_data)
                    tarjan_data.low[u] = min(tarjan_data.low[u], tarjan_data.low[v])
                elif v in tarjan_data.on_stack:
                    tarjan_data.low[u] = min(tarjan_data.low[u], tarjan_data.disc[v])

        if tarjan_data.low[u] == tarjan_data.disc[u]:
            scc = []
            while True:
                node = tarjan_data.stack.pop()
                tarjan_data.on_stack.remove(node)
                scc.append(node)
                if node == u:
                    break
            # Sort within scc for stable output order
            tarjan_data.sccs.append(tuple(reversed(scc)))

    def find_sccs_tarjan(self) -> List[List[str]]:
        """Use Tarjan's algorithm to find all strongly connected components"""
        tarjan_data = self._TarjanDataStruct()
        
        # Sort nodes to ensure algorithm stability
        for node in sorted(self.nodes.keys()):
            if node not in tarjan_data.disc:
                self._tarjan_dfs(node, tarjan_data)
                
        return tarjan_data.sccs
    def get_scc_by_node(self, node: str) -> str:
        """Get the SCC that contains the specified node"""
        for scc_id, nodes in self._scc_nodes.items():
            if node in nodes:
                return scc_id
        raise ValueError(f"Node {node} not found in any SCC")

    def get_scc_edges(self, scc_id: str, source_node: str) -> List[DiGraphEdge]:
        """Get edges within the specified SCC
        
        Args:
            scc_id: Unique identifier of the SCC
            
        Returns:
            List of edges within the SCC
        """
        return self._scc_edges[scc_id][source_node]

    def get_scc_nodes(self, scc_id: str) -> List[str]:
        """Get the list of nodes in the specified SCC
        
        Args:
            scc_id: Unique identifier of the SCC
            
        Returns:
            List of nodes contained in the SCC
        """
        return self._scc_nodes[scc_id]
    def get_leaf_nodes(self) -> List[str]:
        """Get the list of leaf nodes in the graph"""
        start_scc = [scc_id for scc_id, degree in self._scc_in_degree.items() if degree == 0]
        leaf_nodes = []
        for scc_id in start_scc:
            for node in self._scc_nodes[scc_id]:
                if self._scc_nodes_in_degree[scc_id][node] == 0:
                    leaf_nodes.append(node)
        return leaf_nodes

    def get_activation(self, node_name: str) -> Literal["any", "all"]:
        """Get the activation method of the SCC"""
        return self.nodes[node_name].activation

    def get_scc_adj(self) -> Dict[str, List[str]]:
        """Get adjacency relationships between SCCs"""
        return self._scc_adj

    def get_start_nodes(self) -> List[str]:
        """Get nodes with in-degree 0 as starting points"""
        starts_nodes = []
        self.default_start_scc = self.get_scc_by_node(self.default_start_node)
        start_scc = [scc_id for scc_id, degree in self._scc_in_degree.items() if degree == 0]
        for scc_id in start_scc:
            for nodes_in_degree in self._scc_nodes_in_degree[scc_id]:
                for node, in_degree in nodes_in_degree.items():
                    if in_degree == 0:
                        starts_nodes.append(node)
                        #set the default start scc to the first start node found
                        if not self.default_start_scc :
                            self.default_start_scc = scc_id
        if not starts_nodes:
            return [self.default_start_node] if self.default_start_node else []
        else:
            return starts_nodes

    def get_execute_order(self) -> List[str]:
        """Get the execution order of SCCs (topological sort)"""
        # Copy in-degree information to avoid modifying original data
        in_degree = {scc_id: degree for scc_id, degree in self._scc_in_degree.items()}
        queue = deque([scc_id for scc_id, degree in in_degree.items() if degree == 0])
        scc_exec_order = []
        
        while queue:
            scc_id = queue.popleft()
            scc_exec_order.append(scc_id)
            
            for neighbor_scc in self._scc_adj[scc_id]:
                in_degree[neighbor_scc] -= 1
                if in_degree[neighbor_scc] == 0:
                    queue.append(neighbor_scc)
                    
        return scc_exec_order

    def graph_validate(self) -> None:
        """Validate graph structure and execution rules."""
        if not self.nodes:
            raise ValueError("Graph has no nodes.")

        if not self.get_start_nodes():
            raise ValueError("Graph must have at least one start node")

        if not self.get_leaf_nodes():
            raise ValueError("Graph must have at least one leaf node")

        # Outgoing edge condition validation (per node)
        for node in self.nodes.values():
            # Check that if a node has an outgoing conditional edge, then all outgoing edges are conditional
            has_condition = any(
                edge.condition is not None or edge.condition_function is not None for edge in node.edges
            )
            has_unconditioned = any(edge.condition is None and edge.condition_function is None for edge in node.edges)
            if has_condition and has_unconditioned:
                raise ValueError(f"Node '{node.name}' has a mix of conditional and unconditional edges.")

        self._has_cycles = self.has_cycles_with_exit()

class GraphFlowManagerState(BaseGroupChatManagerState):
    """Tracks active execution state for DAG-based execution."""

    active_nodes: List[str] = []  # Currently executing nodes
    type: str = "GraphManagerState"


class GraphFlowManager(BaseGroupChatManager):
    """Manages execution of agents using a Directed Graph execution model."""

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        graph: DiGraphSCC,
    ) -> None:
        """Initialize the graph-based execution manager."""
        super().__init__(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            max_turns=max_turns,
            message_factory=message_factory,
        )
        graph.graph_validate()
        if graph.get_has_cycles() and self._termination_condition is None and self._max_turns is None:
            raise ValueError("A termination condition is required for cyclic graphs without a maximum turn limit.")
        self._graph = graph
        # Lookup table for incoming edges for each node.
        #self._parents = graph.get_parents()

        self._current_scc : List[str]  = []
        # Lookup table for outgoing edges for each node.
        #self._edges: Dict[str, List[DiGraphEdge]] = {n: node.edges for n, node in graph.nodes.items()}
        # Activation lookup table for each node.
        scc_ids : List[str] = graph.get_scc_adj().keys()
        self._activation: Dict[str, Dict[str, Literal["any", "all"]]] = graph.get_scc_nodes_activation()

        # === Mutable states for the graph execution ===
        # Count the number of remaining parents to activate each node.
        self._remaining: Dict[str,Dict[str,Counter[str]]] = graph.get_scc_nodes_in_degree()
        # Lookup table for nodes that have been enqueued through an any activation.
        # This is used to prevent re-adding the same node multiple times.
        self._enqueued_any: Dict[str, Dict[str, bool]] = {
            scc_id: {node: False for node in graph.get_scc_nodes_in_degree()[scc_id]} for scc_id in scc_ids
        }
        # Ready queue for nodes that are ready to execute, starting with the start nodes.
        self._ready: Deque[str] = deque(graph.get_start_nodes())

    def pair_request_with_response(self, message: GraphFlowGroupChatRequestPublish, response: Response, agent_name: str) -> GraphFlowGroupChatAgentResponse:
        """Pair a request with a response."""
        response : GraphFlowGroupChatAgentResponse = GraphFlowGroupChatAgentResponse(agent_response=response, agent_name=agent_name)
        response.agent_response.chat_message.metadata["scc"] = self._current_scc
        return response
    
    def fetch_group_chat_request_publish(self) -> GraphFlowGroupChatRequestPublish:
        """Fetch the group chat request publish from the message."""
        return GraphFlowGroupChatRequestPublish(scc=self._current_scc)

    async def update_message_thread(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> None:
        await super().update_message_thread(messages)

        # Find the node that ran in the current turn.
        message = messages[-1]
        if message.source not in self._graph.nodes:
            # Ignore messages from sources outside of the graph.
            return
        assert isinstance(message, BaseChatMessage)
        source_scc_id :str = message.metadata["scc"]
        source_node = message.source
        self._current_scc = source_scc_id

        # Propagate the update to the children of the node.
        for edge in self._graph.get_scc_edges(source_scc_id, source_node):
            # Use the new check_condition method that handles both string and callable conditions
            if not edge.check_condition(message):
                continue
            target_scc_id = self._graph.get_scc_by_node(edge.target)
            if self._activation[edge.target] == "all":
                self._remaining[target_scc_id][source_node][edge.target] -= 1
                if self._remaining[target_scc_id][source_node][edge.target] == 0:
                    # If all parents are done, add to the ready queue.
                    self._ready.append(edge.target)
                    self._current_scc = target_scc_id
            else:
                # If activation is any, add to the ready queue if not already enqueued.
                if not self._enqueued_any[target_scc_id][edge.target]:
                    self._ready.append(edge.target)
                    self._enqueued_any[target_scc_id][edge.target] = True
                    self._current_scc = target_scc_id
            

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
        # Drain the ready queue for the next set of speakers.
        #switch current_scc
        speakers: List[str] = []
        while self._ready:
            speaker = self._ready.popleft()
            speakers.append(speaker)
            # Reset the bookkeeping for the node that were selected.
            scc_id = self._current_scc
            if self._activation[scc_id][speaker] == "any":
                self._enqueued_any[scc_id][speaker] = False
            else:
                self._remaining[scc_id][speaker] = self._graph.get_scc_nodes_in_degree(scc_id)[speaker]

        # If there are no speakers, trigger the stop agent.
        if not speakers:
            speakers = [_DIGRAPH_STOP_AGENT_NAME]

        return speakers

    async def validate_group_state(self, messages: List[BaseChatMessage] | None) -> None:
        pass

    async def save_state(self) -> Mapping[str, Any]:
        """Save the execution state."""
        state = {
            "message_thread": [message.dump() for message in self._message_thread],
            "current_turn": self._current_turn,
            "remaining": dict(self._remaining),
            "enqueued_any": dict(self._enqueued_any),
            "ready": list(self._ready),
        }
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore execution state from saved data."""
        self._message_thread = [self._message_factory.create(msg) for msg in state["message_thread"]]
        self._current_turn = state["current_turn"]
        self._remaining = Counter(state["remaining"])
        self._enqueued_any = state["enqueued_any"]
        self._ready = deque(state["ready"])

    async def reset(self) -> None:
        """Reset execution state to the start of the graph."""
        self._current_turn = 0
        self._message_thread.clear()
        if self._termination_condition:
            await self._termination_condition.reset()
        self._remaining = Counter({n: len(p) for n, p in self._parents.items()})
        self._enqueued_any = {n: False for n in self._graph.nodes}
        self._ready = deque([n for n in self._graph.get_start_nodes()])


class _StopAgent(BaseChatAgent):
    def __init__(self) -> None:
        super().__init__(_DIGRAPH_STOP_AGENT_NAME, "Agent that terminates the GraphFlow.")

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage, StopMessage)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        return Response(chat_message=StopMessage(content=_DIGRAPH_STOP_AGENT_MESSAGE, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class GraphFlowConfig(BaseModel):
    """The declarative configuration for GraphFlow."""

    participants: List[ComponentModel]
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    graph: DiGraphSCC  # The execution graph for agents


class GraphFlow(BaseGroupChat, Component[GraphFlowConfig]):
    """A team that runs a group chat following a Directed Graph execution pattern.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    This group chat executes agents based on a directed graph (:class:`DiGraph`) structure,
    allowing complex workflows such as sequential execution, parallel fan-out,
    conditional branching, join patterns, and loops with explicit exit conditions.

    The execution order is determined by the edges defined in the `DiGraph`. Each node
    in the graph corresponds to an agent, and edges define the flow of messages between agents.
    Nodes can be configured to activate when:

        - **All** parent nodes have completed (activation="all") → default
        - **Any** parent node completes (activation="any")

    Conditional branching is supported using edge conditions, where the next agent(s) are selected
    based on content in the chat history. Loops are permitted as long as there is a condition
    that eventually exits the loop.

    .. note::

        Use the :class:`DiGraphBuilder` class to create a :class:`DiGraph` easily. It provides a fluent API
        for adding nodes and edges, setting entry points, and validating the graph structure.
        See the :class:`DiGraphBuilder` documentation for more details.
        The :class:`GraphFlow` class is designed to be used with the :class:`DiGraphBuilder` for creating complex workflows.

    .. warning::

        When using callable conditions in edges, they will not be serialized
        when calling :meth:`dump_component`. This will be addressed in future releases.


    Args:
        participants (List[ChatAgent]): The participants in the group chat.
        termination_condition (TerminationCondition, optional): Termination condition for the chat.
        max_turns (int, optional): Maximum number of turns before forcing termination.
        graph (DiGraph): Directed execution graph defining node flow and conditions.

    Raises:
        ValueError: If participant names are not unique, or if graph validation fails (e.g., cycles without exit).

    Examples:

        **Sequential Flow: A → B → C**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent("A", model_client=model_client, system_message="You are a helpful assistant.")
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to Chinese.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to English.")

                # Create a directed graph with sequential flow A -> B -> C.
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b).add_edge(agent_b, agent_c)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short story about a cat."):
                    print(event)


            asyncio.run(main())

        **Parallel Fan-out: A → (B, C)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent("A", model_client=model_client, system_message="You are a helpful assistant.")
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to Chinese.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to Japanese.")

                # Create a directed graph with fan-out flow A -> (B, C).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b).add_edge(agent_a, agent_c)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short story about a cat."):
                    print(event)


            asyncio.run(main())

        **Conditional Branching: A → B (if 'yes') or C (otherwise)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent(
                    "A",
                    model_client=model_client,
                    system_message="Detect if the input is in Chinese. If it is, say 'yes', else say 'no', and nothing else.",
                )
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to English.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to Chinese.")

                # Create a directed graph with conditional branching flow A -> B ("yes"), A -> C (otherwise).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                # Create conditions as callables that check the message content.
                builder.add_edge(agent_a, agent_b, condition=lambda msg: "yes" in msg.to_model_text())
                builder.add_edge(agent_a, agent_c, condition=lambda msg: "yes" not in msg.to_model_text())
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="AutoGen is a framework for building AI agents."):
                    print(event)


            asyncio.run(main())

        **Loop with exit condition: A → B → C (if 'APPROVE') or A (otherwise)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1")
                agent_a = AssistantAgent(
                    "A",
                    model_client=model_client,
                    system_message="You are a helpful assistant.",
                )
                agent_b = AssistantAgent(
                    "B",
                    model_client=model_client,
                    system_message="Provide feedback on the input, if your feedback has been addressed, "
                    "say 'APPROVE', otherwise provide a reason for rejection.",
                )
                agent_c = AssistantAgent(
                    "C", model_client=model_client, system_message="Translate the final product to Korean."
                )

                # Create a loop graph with conditional exit: A -> B -> C ("APPROVE"), B -> A (otherwise).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b)

                # Create conditional edges using strings
                builder.add_edge(agent_b, agent_c, condition=lambda msg: "APPROVE" in msg.to_model_text())
                builder.add_edge(agent_b, agent_a, condition=lambda msg: "APPROVE" not in msg.to_model_text())

                builder.set_entry_point(agent_a)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(20),  # Max 20 messages to avoid infinite loop.
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short poem about AI Agents."):
                    print(event)


            asyncio.run(main())
    """

    component_config_schema = GraphFlowConfig
    component_provider_override = "autogen_agentchat.teams.GraphFlow"

    def __init__(
        self,
        participants: List[ChatAgent],
        graph: DiGraph,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
    ) -> None:
        self._input_participants = participants
        self._input_termination_condition = termination_condition

        stop_agent = _StopAgent()
        stop_agent_termination = StopMessageTermination()
        termination_condition = (
            stop_agent_termination
            if not termination_condition
            else OrTerminationCondition(stop_agent_termination, termination_condition)
        )

        participants = [stop_agent] + participants
        super().__init__(
            participants,
            group_chat_manager_name="GraphManager",
            group_chat_manager_class=GraphFlowManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
        )
        self._graph = graph

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
    ) -> Callable[[], GraphFlowManager]:
        """Creates the factory method for initializing the DiGraph-based chat manager."""

        def _factory() -> GraphFlowManager:
            return GraphFlowManager(
                name=name,
                group_topic_type=group_topic_type,
                output_topic_type=output_topic_type,
                participant_topic_types=participant_topic_types,
                participant_names=participant_names,
                participant_descriptions=participant_descriptions,
                output_message_queue=output_message_queue,
                termination_condition=termination_condition,
                max_turns=max_turns,
                message_factory=message_factory,
                graph=self._graph,
            )

        return _factory

    def _to_config(self) -> GraphFlowConfig:
        """Converts the instance into a configuration object."""
        participants = [participant.dump_component() for participant in self._input_participants]
        termination_condition = (
            self._input_termination_condition.dump_component() if self._input_termination_condition else None
        )
        return GraphFlowConfig(
            participants=participants,
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            graph=self._graph,
        )

    @classmethod
    def _from_config(cls, config: GraphFlowConfig) -> Self:
        """Reconstructs an instance from a configuration object."""
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )
        return cls(
            participants, graph=config.graph, termination_condition=termination_condition, max_turns=config.max_turns
        )
