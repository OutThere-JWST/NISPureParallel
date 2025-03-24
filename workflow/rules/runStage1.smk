# High memory fields
high_memory_fields = ['boo-04','leo-02','leo-11','leo-13','sex-04','sex-15','sex-16','sex-17','sex-22','sex-29','uma-02','vir-04']

# Compute resources for all fields
cpus_per_task = workflow.resource_settings.default_resources.parsed['cpus_per_task']
resources = {
    field: max(1, cpus_per_task // 2) if field in high_memory_fields else cpus_per_task
    for field in FIELDS
}


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
        resources:
            cpus_per_task = resources[field]
        shell: 
            """
            mkdir -p FIELDS/{field}/RATE
            pixi run --no-lockfile-update --environment jwst parallel --link -j {resources.cpus_per_task} ./workflow/scripts/runStage1.py --scratch ::: {input} ::: {output} > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)
