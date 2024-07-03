# Redshift Fitting Rule
rule zfit:
    input:
        'logs/{field}-extr.log'
    output:
        'logs/{field}-zfit.log'
    log:
        'logs/{field}-zfit.log'
    group:
        lambda wildcards: f'zfit-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    shell:
        """
        ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {threads} &> {log}
        """