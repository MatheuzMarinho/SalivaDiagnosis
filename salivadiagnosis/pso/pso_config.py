class PsoConfig:
    def __init__(self, config_content):
        config_list = config_content.split('\n')
        self.population_size = int(config_list[0])
        self.max_iterations = int(config_list[1])
        self.cognitive_coeff = float(config_list[2])
        self.social_coeff = float(config_list[3])
        self.inertia_coeff = str(config_list[4])
