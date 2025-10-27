import itertools
from dataclasses import dataclass
from functools import cached_property
from math import inf
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial

from analysis import GelProxy
from utils import mymath


def graph_from_beads(
    bead_ids: np.ndarray, crosslinks: np.ndarray, chain_length: int
) -> nx.Graph:
    G = nx.Graph()

    for chain in np.split(bead_ids, len(bead_ids) // chain_length):
        for i, j in itertools.pairwise(chain):
            assert i != j, (i, j)
            assert i == j - 1, (i, j)
            G.add_edge(i, j, edge_type="chain")

    for i, j in crosslinks:
        assert i < j, (i, j)
        # assert j - i > 1, (i, j)  # this is not true because of chain ends
        G.add_edge(i, j, edge_type="crosslink")

    return G


def graph_from_gel_proxy(gel: GelProxy) -> nx.Graph:
    G = nx.Graph()

    for i in range(gel.n_beads):
        G.add_node(i)

    for i, j in gel.chain_pairs:
        assert i != j, (i, j)
        assert i == j - 1, (i, j)
        G.add_edge(i, j, edge_type="chain")

    for i, j in gel.crosslinks:
        assert i < j, (i, j)
        # assert j - i > 1, (i, j)
        G.add_edge(i, j, edge_type="crosslink")

    return G


@dataclass(frozen=True)
class PoreAnalyzer:
    graph: nx.Graph

    @classmethod
    def from_gel_dir(cls, gel_dir_path: Path):
        return cls(graph_from_gel_proxy(GelProxy.from_gel_dir(gel_dir_path)))

    @classmethod
    def from_bonds(cls, bond_pairs: Iterable[Sequence[int]]) -> "PoreAnalyzer":
        G = nx.Graph()
        for i, j in bond_pairs:
            G.add_edge(i, j)
        return cls(G)

    @cached_property
    def pores(self) -> list[list[int]]:
        return nx.cycle_basis(self.graph)

    @cached_property
    def beads_per_pore(self) -> np.ndarray:
        return np.array([len(p) for p in self.pores])

    def pore_areas(self, bead_points: np.ndarray):
        for pore in self.pores:
            yield self._approximate_pore_area(bead_points[pore])

    @staticmethod
    def _approximate_pore_area(pore_bead_points: np.ndarray):
        """Fit plane to points -> project points on plane -> compute (convex) polygon area"""
        res = mymath.project_points_onto_best_plane(pore_bead_points)
        assert res.projected_points.shape[1] == 2
        return spatial.ConvexHull(res.projected_points).volume


def calculate_distances_from_edge_type(
    G: nx.Graph, edge_type: str, dist_to_edge_attr_name: Optional[str] = None
):
    if dist_to_edge_attr_name is None:
        dist_to_edge_attr_name = f"dist_to_{edge_type}"

    edge_type_nodes = set(
        n for edge, d in G.edges.items() if d["edge_type"] == edge_type for n in edge
    )

    for n in G.nodes:
        G.nodes[n][dist_to_edge_attr_name] = inf

    for cn in edge_type_nodes:
        G.nodes[cn][dist_to_edge_attr_name] = 0
        node_stack = [(n, cn) for n in G.neighbors(cn)]
        while len(node_stack) > 0:
            n, parent = node_stack.pop()

            if n in edge_type_nodes:
                continue

            G.nodes[n][dist_to_edge_attr_name] = min(
                G.nodes[parent][dist_to_edge_attr_name] + 1,
                G.nodes[n][dist_to_edge_attr_name],
            )

            node_stack.extend((child, n) for child in G.neighbors(n) if child != parent)


def _can_node_be_removed(G: nx.Graph, node) -> tuple[bool, Any]:
    if G.degree[node] != 2:  # type: ignore
        return False, None

    neighbors = tuple(G.neighbors(node))
    for neighbor_node in neighbors:
        # if one neighbor has degree < 3, it can be merged with that neighbor
        if G.degree[neighbor_node] < 3:  # type: ignore
            if neighbors not in G.edges:
                return True, neighbor_node
    else:
        # if not, it can still be merged, if the edge resulting from that not already exists
        assert len(neighbors) == 2, len(neighbors)
        if neighbors not in G.edges:
            return True, neighbor_node

    return False, None


def linearize_graph(G: nx.Graph):
    G2 = G.copy()

    nodes_to_remove = list(G2.nodes)
    n_removed = 0
    while len(nodes_to_remove) > 0:
        node = nodes_to_remove.pop()
        can_remove, neighbor = _can_node_be_removed(G2, node)
        if can_remove:
            G2 = nx.contracted_nodes(G2, neighbor, node, self_loops=False)
            n_removed += 1

    if n_removed > 0:
        G2 = linearize_graph(G2)

    assert len(nx.cycle_basis(G)) == len(nx.cycle_basis(G2)), f"{len(nx.cycle_basis(G))=} != {len(nx.cycle_basis(G2))=}"

    return G2


def remove_dangling_ends(G: nx.Graph):
    G2 = G.copy()

    for node in G.nodes:
        if G2.degree[node] == 1:  # type: ignore
            G2.remove_node(node)
            break
    else:
        return G2

    return remove_dangling_ends(G2)


if __name__ == "__main__":
    # test `linearize_graph` function
    fig, axs = plt.subplots(ncols=2, nrows=10, figsize=(8, 20))
    axs = axs.flat

    PLOT_KWARGS = dict(
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        edge_color="gray",
        font_size=10,
    )

    def plot_graph(G: nx.Graph, ax: plt.Axes):
        nx.draw(G, ax=ax, **PLOT_KWARGS)  # type: ignore

    G = nx.Graph(
        [(1, 2), (1, 3), (2, 3), (2, 7), (3, 5), (4, 6), (4, 5), (7, 4), (6, 9)]
    )
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(1, 2), (2, 3), (3, 4), (2, 5)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(1, 2), (2, 3), (3, 2)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 2)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.cycle_graph(6)
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.star_graph(4)
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(0, 1), (1, 2), (2, 3), (1, 3)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    G = nx.Graph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (0, 5)])
    plot_graph(G, ax=next(axs))
    plot_graph(linearize_graph(G), ax=next(axs))

    plt.tight_layout()
    plt.show()
