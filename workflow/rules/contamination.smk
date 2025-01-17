# Contamination Rule
rule contam:
    input:
        'FIELDS/{field}/logs/mos.log'
    output:
        'FIELDS/{field}/logs/contam.log'
    log:
        'FIELDS/{field}/logs/contam.log'
    group:
        lambda wildcards: f'contam-{groups[wildcards.field]}'
    resources:
        slurm_extra = lambda wildcards: f'-J contam-{groups[wildcards.field]}'
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/contamination.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """