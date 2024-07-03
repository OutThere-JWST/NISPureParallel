# Redshift Fitting Rule
rule zfit:
    input:
        'FIELDS/{field}/logs/extr.out'
    output:
        'FIELDS/{field}/logs/zfit.out'
    conda:'../envs/grizli.yaml'
    log:
        stdout='logs/{field}-zfit.out',
        stderr='logs/{field}-zfit.err'
    shell:
        """
        ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {threads} > {log.stdout} 2> {log.stderr}
        """