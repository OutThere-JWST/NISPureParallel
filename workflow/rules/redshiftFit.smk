# Redshift Fitting Rule
rule zfit:
    input:
        'FIELDS/{field}/logs/extr.log'
    output:
        'FIELDS/{field}/logs/zfit.log'
    log:
        'FIELDS/{field}/logs/zfit.log'
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """