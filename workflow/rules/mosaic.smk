# Mosaic Rule
rule mosaic:
    input:
        'logs/{field}-proc.log'
    output:
        'logs/{field}-mos.log'
    conda:
        '../envs/grizli.yaml'
    log:
        'logs/{field}-mos.log'
    shell:
        """
        ./workflow/scripts/mosaic.py {wildcards.field} &> {log}
        """