# Mosaic Rule
rule mosaic:
    input:
        'FIELDS/{field}/logs/proc.log'
    output:
        'FIELDS/{field}/logs/mos.log'
    log:
        'FIELDS/{field}/logs/mos.log'
    group:
        lambda wildcards: f'mos-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/mosaic.py {wildcards.field} > {log} 2>&1
        """