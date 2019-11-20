#   imports
from map_elites import map_elites
archive = map_elites()
archive.feature_space.add_feature('avg clustering', 0, 1, 20)
archive.feature_space.add_feature('avg shortest path', 1, 3, 20)
archive.run(50, 1000, 'CartPole-v0')
