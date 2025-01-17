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
            lambda wildcards: f'fmap-{groups[field]}'
        priority:
            priority
        resources:
            # slurm_extra = f'-J fmap-{field}'
            # tasks = lambda wildcards: len(uncal[wildcards.field])
        shell:
            f"""
            pixi run --no-lockfile-update --environment fitsmap ./workflow/scripts/makeFitsmap.py {field} --ncpu {{resources.cpus_per_task}} > {{log}} 2>&1
            """
            # tar -cf FIELDS/{wildcards.field}/fitsmap/{wildcards.field}.tar -C FIELDS/{wildcards.field}/fitsmap/ {wildcards.field}/


# Create rules for all fields
for field in FIELDS: create_rule(field)