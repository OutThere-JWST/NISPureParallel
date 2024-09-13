# Fitsmap Creation Rule
def create_rule(field):
    priority = len(uncal[field])
    rule:
        name: f"fmap_{field}"
        input:
            f'FIELDS/{field}/logs/zfit.log'
        output:
            f'FIELDS/{field}/logs/fmap.log'
        log:
            f'FIELDS/{field}/logs/fmap.log'
        group:
            lambda wildcards: f'fmap-{groups[wildcards.field]}'
        conda:
            '../envs/fitsmap.yaml'
        priority:
            priority
        # resources:
        #     tasks = lambda wildcards: len(uncal[wildcards.field])
        shell:
            """
            ./workflow/scripts/makeFitsmap.py {field} --ncpu {resources.cpus_per_task} > {log} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)