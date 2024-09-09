# Redshift Fitting Rule
rule zfit:
    input:
        'FIELDS/{field}/logs/extr.log'
    output:
        'FIELDS/{field}/logs/zfit.log'
    log:
        'FIELDS/{field}/logs/zfit.log'
    group:
        lambda wildcards: f'zfit-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """