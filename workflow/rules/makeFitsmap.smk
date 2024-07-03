# Fitsmap Creation Rule
rule fmap:
    input:
        'FIELDS/{field}/logs/zfit.out'
    output:
        'FIELDS/{field}/logs/fmap.out'
    conda:'../envs/fitsmap.yaml'
    log:
        stdout='logs/{field}-fmap.out',
        stderr='logs/{field}-fmap.err'
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --slowsegmap --ncpu {threads} > {log.stdout} 2> {log.stderr}
        """