# High memory fields
high_memory_fields = ['vir-04','leo-11','boo-04','uma-02','leo-02']

# Stage 1 Rule
def create_rule(field):

    rule:
        name: f"stage1_{field}"
        input:
            [f'FIELDS/{field}/UNCAL/{f}' for f in uncal[field]]
        output:
            [f'FIELDS/{field}/RATE/{f.replace('uncal','rate')}' for f in uncal[field]]
        log: 
            f'FIELDS/{field}/logs/stage1.log'
        group:
            f'stage1-{groups[field]}'
        conda:
            '../envs/jwst.yaml'
        resources:
            cpus_per_task = lambda wildcards: 1 if field in high_memory_fields else config['default-resources']['cpus_per_task']
        shell: 
            """
            mkdir -p FIELDS/{field}/RATE
            parallel -j {resources.cpus_per_task} ./workflow/scripts/runStage1.py --scratch ::: {input} ::: {output} > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)
