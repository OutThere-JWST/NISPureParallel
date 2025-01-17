# High memory fields
high_memory_fields = ['boo-04','leo-02','leo-11','leo-13','sex-04','sex-16','sex-17','sex-22','sex-29','uma-02','vir-04']

# Stage 1 Rule
def create_rule(field):

    # Get default resources and scale CPUs down by half
    default_resources = workflow.resource_settings.default_resources.parsed.copy()
    default_resources['cpus_per_task'] = max(1, default_resources['cpus_per_task'] // 2)

    # Replace resources
    if field in high_memory_fields:
        default_resources['cpus_per_task'] = 2

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
            cpus_per_task = default_resources['cpus_per_task'],
            slurm_extra = f'-J stage1-{field}'
        shell: 
            """
            mkdir -p FIELDS/{field}/RATE
            pixi run --no-lockfile-update --environment jwst parallel --link -j {resources.cpus_per_task} ./workflow/scripts/runStage1.py --scratch ::: {input} ::: {output} > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)
