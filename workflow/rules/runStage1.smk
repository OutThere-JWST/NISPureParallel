# Download Rule
rule stage1:
    input:
        'FIELDS/{field}/UNCAL/{file}_uncal.fits'
    output:
        'FIELDS/{field}/RATE/{file}_rate.fits'
    log: 
        'FIELDS/{field}/logs/files/{file}.log'
    group:
        'stage1'
    # resources:
    #     mem_mb = lambda _, input: 5 * input.size_mb
    shell: 
        """
        ./workflow/scripts/download.py {input} > {log} 2>&1
        pixi run --no-lockfile-update --environment jwst crds sync --contexts $CRDS_CONTEXT --fetch-references --dataset-files {input} >> {log} 2>&1
        """