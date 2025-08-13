# Download Rule
rule download:
    output:
        'FIELDS/{field}/UNCAL/{file}_uncal.fits'
    log: 
        'FIELDS/{field}/logs/files/{file}.log'
    group:
        'download'
    resources:
    shell: 
        """
        ./workflow/scripts/download.py {output} > {log} 2>&1
        pixi run --no-lockfile-update --environment jwst crds sync --contexts $CRDS_CONTEXT --fetch-references --dataset-files {input} >> {log} 2>&1
        """