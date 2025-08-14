# Preprocess Rule
rule preprocess:
    input:
        lookup(dpath='{field}', within=rate)
    output:
        'FIELDS/{field}/logs/proc.log'
    log:
        'FIELDS/{field}/logs/proc.log'
    group:
        'imaging'
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/preprocess.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """