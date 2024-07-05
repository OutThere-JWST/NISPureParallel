# Extraction Rule
rule extract:
    input:
        'logs/{field}-contam.log'
    output:
        'logs/{field}-extr.log'
    log:
        'logs/{field}-extr.log'
    group:
        lambda wildcards: f'extr-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/extract.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """