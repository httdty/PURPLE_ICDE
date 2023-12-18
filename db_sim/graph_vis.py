import networkx as nx
import matplotlib.pyplot as plt
import dill

with open("./datasets/spider/schema_graph.bin", 'rb') as f:
    schema_graph = dill.load(f)

graph_node = []
G_list = {}

for i in schema_graph:
    G = nx.Graph()
    node_colors = []
    edge_colors = []
    # tmp = []
    G.add_edges_from(i["graph"][0])
    G.add_nodes_from(i["graph"][1])

    for j in G.nodes():

        if i["graph"][1][j] == 0:
            node_colors.append('#2F4F4F')
        elif i["graph"][1][j] == 1:
            node_colors.append('#E6E6FA')
        elif i["graph"][1][j] == 2:
            node_colors.append('#D8BFD8')
        elif i["graph"][1][j] == 3:
            node_colors.append('#FFC0CB')
    for e in G.edges:
        if i["graph"][2][e] == 4:
            edge_colors.append('#DC143C')
        else:
            edge_colors.append('#000000')

    pic = plt.plot()
    plt.title(i['schema']['db_id'])
    # limits = plt.axis("off")
    nx.draw(G, pos=nx.spring_layout(G),with_labels=False, font_weight='bold', node_color=node_colors,edge_color=edge_colors)
    # G_list[i['schema']['db_id']] = pic
    plt.savefig(f"datasets/db_vis/{i['schema']['db_id']}.png", dpi=300)
    plt.show()

