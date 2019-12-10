class GaConfig:
    def __init__(self, config_content):
        config_list = config_content.split('\n')
        self.num_individuals = int(config_list[0])
        self.mutation_ratio = float(config_list[1])
        self.selection_type = int(config_list[2])
        self.num_descendants = int(config_list[3])
