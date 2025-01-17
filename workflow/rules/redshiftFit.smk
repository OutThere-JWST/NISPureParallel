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
    resources:
        # tasks = lambda wildcards: len(uncal[wildcards.field])
        # slurm_extra = lambda wildcards: f'-J zfit-{groups[wildcards.field]}'
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """