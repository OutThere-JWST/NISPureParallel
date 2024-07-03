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
    shell:
        """
        ./workflow/scripts/mosaic.py {wildcards.field} &> {log}
        """