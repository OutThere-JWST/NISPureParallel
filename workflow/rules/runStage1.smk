# Stage 1 Rule
def create_rule(field):
    rule:
        name: f"stage1_{field}"
        input:
            [f'UNCAL/{f}' for f in uncal[field]]
        output:
            [f'RATE/{f.replace('uncal','rate')}' for f in uncal[field]]
        conda:'../envs/jwst.yaml'
        log: 
            stdout = f'logs/{field}-stage1.log',
            stderr = f'logs/{field}-stage1.err'
        shell: 
            f"""
            ./workflow/scripts/runStage1.py {field} --ncpu {{threads}} > {{log.stdout}} 2> {{log.stderr}}
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)