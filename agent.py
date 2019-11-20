from bindsnet.network import Network
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.environment import GymEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_softmax
from genotype import genotype
import networkx as nx
from matplotlib import pyplot as plt

class agent:
    '''
        agent class for acting in the openAI gym environments
    '''
    def __init__(self, gt=None):
        if gt is None:
            self.genotype = genotype(p_max=20)
        else:
            self.genotype = gt

    def phenotype(self, adj_mat):
        '''
        express phenotype as networkx Graph and then measure
        the properties of the largest connected component
        '''
        self.G = nx.Graph(adj_mat.numpy())
        max_cc = max(nx.connected_components(self.G), key=len)
        self.Gc = nx.subgraph(self.G, max_cc)
        b = {'avg clustering': nx.average_clustering(self.Gc),
             'avg shortest path': nx.average_shortest_path_length(self.Gc)}
        return b

    def draw_graph(self):
        nx.draw(self.G, pos=nx.random_layout(self.G))
        plt.plot()
        plt.show()

    def evaluate(self, env):
        '''
            evaluates the agent on a given environment
        '''

        #   build the network
        network = Network(dt=1.0)
        #   layers of neurons
        layer_names = ['S', 'I', 'M']
        layers = [Input(n=self.genotype.sz[0], traces=True),
                  LIFNodes(n=self.genotype.sz[1], traces=True),
                  LIFNodes(n=self.genotype.sz[2], refrac=0, traces=True)]
        for i,l in enumerate(layers):
            network.add_layer(l, layer_names[i])

        #   connections
        c_dict = self.genotype.express()
        for i,source in enumerate(layers):
            for j,target in enumerate(layers):
                source_name = layer_names[i]
                target_name = layer_names[j]
                name = layer_names[i]+layer_names[j]
                connection = Connection(source, target, w=c_dict[name])
                network.add_connection(connection, source=source_name, target=target_name)

        #   create pipeline object for interfacing with openAI gym
        environment = GymEnvironment(env)
        environment.reset()
        pipeline = EnvironmentPipeline(
            network,
            environment,
            encoding=bernoulli,
            action_function=select_softmax,
            output="M",
            time=1,
            history_length=1,
            delta=1,
            plot_interval=1,
            #render_interval=1,
        )

        #   analyze agent network attributes
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        while not is_done:
            result = pipeline.env_step()
            pipeline.step(result)

            reward = result[1]
            total_reward += reward

            is_done = result[2]
        pipeline.env.close()
        print(f"Episode {i} total reward:{total_reward}")
        p = total_reward
        b = self.phenotype(self.genotype.adj_mat)
        return p,b

