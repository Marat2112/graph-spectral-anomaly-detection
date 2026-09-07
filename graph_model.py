import numpy as np
import networkx as nx


def build_graph(
    num_nodes: int,
    graph_type: str = "ring",
    rng=None
):
    """
    Создаёт граф сенсорной сети.

    Поддерживаемые типы:
        path
        ring
        grid
        random_geometric
    """

    if graph_type == "path":

        G = nx.path_graph(
            num_nodes
        )

    elif graph_type == "ring":

        G = nx.cycle_graph(
            num_nodes
        )

    elif graph_type == "grid":

        side = int(
            np.ceil(
                np.sqrt(num_nodes)
            )
        )

        G_full = nx.grid_2d_graph(
            side,
            side
        )

        nodes = list(
            G_full.nodes()
        )[:num_nodes]

        G = G_full.subgraph(
            nodes
        ).copy()

        G = nx.convert_node_labels_to_integers(
            G
        )

    elif graph_type == "random_geometric":

        if rng is None:
            seed = 42
        else:
            seed = int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            )

        radius = 0.35

        G = nx.random_geometric_graph(
            num_nodes,
            radius=radius,
            seed=seed
        )

        # Если граф оказался несвязным,
        # соединяем компоненты минимальным способом.
        components = list(
            nx.connected_components(G)
        )

        if len(components) > 1:

            for c1, c2 in zip(
                components[:-1],
                components[1:]
            ):
                u = next(iter(c1))
                v = next(iter(c2))

                G.add_edge(
                    u,
                    v
                )

    else:

        raise ValueError(
            f"Неизвестный graph_type: {graph_type}"
        )

    return G


def adjacency_matrix(G):

    return nx.to_numpy_array(
        G,
        dtype=float
    )


def laplacian_matrix(G):

    A = adjacency_matrix(G)

    degrees = A.sum(axis=1)

    D = np.diag(
        degrees
    )

    return D - A


def graph_energy(
    x: np.ndarray,
    L: np.ndarray
) -> float:

    return float(
        x.T @ L @ x
    )


def graph_energy_batch(
    X: np.ndarray,
    L: np.ndarray
) -> np.ndarray:

    return np.einsum(
        "bi,ij,bj->b",
        X,
        L,
        X
    )


def graph_spectral_characteristics(
    L: np.ndarray
):
    """
    Основные спектральные характеристики
    лапласиана.
    """

    eigenvalues = np.linalg.eigvalsh(
        L
    )

    eigenvalues = np.sort(
        eigenvalues
    )

    trace_L = np.trace(
        L
    )

    trace_L2 = np.trace(
        L @ L
    )

    lambda_2 = (
        eigenvalues[1]
        if len(eigenvalues) > 1
        else 0.0
    )

    lambda_max = eigenvalues[-1]

    return {
        "trace_L":
            float(trace_L),

        "trace_L2":
            float(trace_L2),

        "lambda_2":
            float(lambda_2),

        "lambda_max":
            float(lambda_max),

        "eigenvalues":
            eigenvalues
    }