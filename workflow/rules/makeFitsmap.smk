# Fitsmap Creation Rule
rule fmap:
    input:
        'logs/{field}-zfit.log'
    output:
        'logs/{field}-fmap.log'
    log:
        'logs/{field}-fmap.log'
    group:
        lambda wildcards: f'fmap-{groups[wildcards.field]}'
    conda:
        '../envs/fitsmap.yaml'
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --slowsegmap --ncpu {threads} &> {log}
        """