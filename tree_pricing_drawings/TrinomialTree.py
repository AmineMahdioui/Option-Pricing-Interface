import networkx as nx
import matplotlib.pyplot as plt

class TrinomialTree:
    def __init__(self, matrix,):
        self.G = nx.DiGraph()
        self.matrix = matrix
        self.labels = {}
        self.pos = None
        self.create_tree()
        self.set_node_labels()
    def create_tree(self):
        rows = len(self.matrix)
        for i in range(rows):
            for j in range(2*i+1):
                parent = (i, j)
                if i+1  < rows:
                    left_child = (i+1, j+2)
                    middle_child = (i+1, j+1)
                    right_child = (i+1, j)
                    self.G.add_edge(parent, right_child)
                    self.G.add_edge(parent, middle_child)
                    self.G.add_edge(parent, left_child)

    def set_node_labels(self):
        for node in self.G.nodes():
            i, j = node
            try:
                self.labels[node] ="$V_{"+str(i)+"}"+f"$={self.matrix[i][j]:.2f}"
            except IndexError:
                self.labels[node] = 'out_of_bounds'

    def draw_tree(self):
        # compute layout; prefer graphviz if available otherwise fallback
        if self.pos is None:
            try:
                self.pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')
            except Exception:
                self.pos = nx.spring_layout(self.G)

        pos_graphviz = {k: (-v[1], -v[0]) for k, v in self.pos.items()}

        scale = 1.0 / max(len(self.matrix), 1)
        fig, ax = plt.subplots(figsize=(15, int(1 / scale)))

        nx.draw(self.G, pos_graphviz, with_labels=True, labels=self.labels, node_size=1000, node_color='#FF0800', font_size=10, font_color='black', ax=ax)
        return fig