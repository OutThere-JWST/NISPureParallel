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
    shell:
        """
        ./workflow/scripts/extract.py {wildcards.field} --ncpu {threads} &> {log}
        """