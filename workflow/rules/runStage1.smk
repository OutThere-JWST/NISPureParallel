# Stage 1 Rule
def create_rule(field):
    rule:
        name: f"stage1_{field}"
        input:
            [f'UNCAL/{f}' for f in uncal[field]]
        output:
            [f'RATE/{f.replace('uncal','rate')}' for f in uncal[field]]
        log: 
            f'logs/{field}-stage1.log'
        group:
            f'stage1-{groups[field]}'
        conda:
            '../envs/jwst.yaml'
        # resources:
        #     tasks = len(uncal[field])
        shell: 
            """
            echo {input} | tr ' ' '\\n' |\\
            parallel -j {resources.cpus_per_task} ./workflow/scripts/runStage1.py --scratch > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)