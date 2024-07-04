# Mosaic Rule
rule mosaic:
    input:
        'logs/{field}-proc.log'
    output:
        'logs/{field}-mos.log'
    log:
        'logs/{field}-mos.log'
    group:
        lambda wildcards: f'mos-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    resources:
        tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/mosaic.py {resources.tasks} &> {log}
        """