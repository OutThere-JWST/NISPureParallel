# Extraction Rule
rule extract:
    input:
        'logs/{field}-contam.log'
    output:
        'logs/{field}-extr.log'
    conda:
        '../envs/grizli.yaml'
    log:
        'logs/{field}-extr.log'
    shell:
        """
        ./workflow/scripts/extract.py {wildcards.field} --ncpu {threads} &> {log}
        """