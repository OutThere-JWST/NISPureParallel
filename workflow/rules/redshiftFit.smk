# Redshift Fitting Rule
rule zfit:
    input:
        'logs/{field}-extr.log'
    output:
        'logs/{field}-zfit.log'
    conda:
        '../envs/grizli.yaml'
    log:
        'logs/{field}-zfit.log'
    shell:
        """
        ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {threads} &> {log}
        """