# Contamination Rule
rule contam:
    input:
        'logs/{field}.mos.log'
    output:
        'logs/{field}.contam.log'
    log:
        'logs/{field}.contam.log'
    group:
        lambda wildcards: f'contam-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/contamination.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """