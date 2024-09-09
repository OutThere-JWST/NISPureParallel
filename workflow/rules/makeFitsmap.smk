# Fitsmap Creation Rule
rule fmap:
    input:
        'FIELDS/{field}/logs/zfit.log'
    output:
        'FIELDS/{field}/logs/fmap.log'
    log:
        'FIELDS/{field}/logs/fmap.log'
    group:
        lambda wildcards: f'fmap-{groups[wildcards.field]}'
    conda:
        '../envs/fitsmap.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        tar -cf FIELDS/{wildcards.field}/fitsmap/{wildcards.field}.tar -C FIELDS/{wildcards.field}/fitsmap/ {wildcards.field}/
        """