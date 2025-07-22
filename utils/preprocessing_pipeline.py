class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step, input_column = None, output_column = None, active=True):
        self.steps.append({'step': step, 'input': input_column, 'output': output_column, 'active': active})

    def process(self, df):
        df_copy = df.copy()
        for step in self.steps:
            if step['active']:
                if step['input'] and step['output']:
                    df_copy[step['output']] = df_copy[step['input']].apply(step['step'])
                else:
                    df_copy = step['step'](df_copy)
        return df_copy

    def set_active(self, step_name, active):
        for step in self.steps:
            if step['step'].__name__ == step_name:
                step['active'] = active
                
    def get_active_steps(self):
        return [step['step'].__name__ for step in self.steps if step['active']]