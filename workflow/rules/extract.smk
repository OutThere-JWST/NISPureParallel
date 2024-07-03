# Extraction Rule
rule extract:
    input:
        'FIELDS/{field}/logs/contam.out'
    output:
        'FIELDS/{field}/logs/extr.out'
    conda:'../envs/grizli.yaml'
    log:
        stdout='logs/{field}-extr.out',
        stderr='logs/{field}-extr.err'
    shell:
        """
        ./workflow/scripts/extract.py {wildcards.field} --ncpu {threads} > {log.stdout} 2> {log.stderr}
        """