# High memory field resources
high_mem_resurces = {field:{'cpus_per_task': 1} for field in ['vir-04','leo-11','boo-04','uma-02','leo-02']}

# Stage 1 Rule
def create_rule(field):

    # Swap in high_memory resources for high memory fields
    default_resources = {'cpus_per_task': '{resources.cpus_per_task}'}
    resources = high_mem_resurces.get(field, default_resources)

    rule:
        name: f"stage1_{field}"
        input:
            [f'FIELDS/{field}/UNCAL/{f}' for f in uncal[field]]
        output:x
            [f'FIELDS/{field}/RATE/{f.replace('uncal','rate')}' for f in uncal[field]]
        log: 
            f'FIELDS/{field}/logs/stage1.log'
        group:
            f'stage1-{groups[field]}'
        conda:
            '../envs/jwst.yaml'
        resources:
            cpus_per_task = resources['cpus_per_task']
        shell: 
            """
            mkdir -p FIELDS/{field}/RATE
            parallel -j {resources.cpus_per_task} ./workflow/scripts/runStage1.py --scratch ::: {input} ::: {output} > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)
