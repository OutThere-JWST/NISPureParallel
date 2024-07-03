# Redshift Fitting Rule
rule zfit:
    input:
        'FIELDS/{field}/logs/extr.out'
    output:
        'FIELDS/{field}/logs/zfit.out'
    conda:'../envs/grizli.yaml'
    log:'logs/Redshift_{field}.log'
    shell:
        """
        ./workflow/scripts/redshiftFit.py {wildcards.field} --ncpu {threads}
        """