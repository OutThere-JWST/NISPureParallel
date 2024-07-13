window.onload = function() {

    const images = document.querySelectorAll('.image img');

    images.forEach(img => {

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 256;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        let minX = canvas.width;
        let minY = canvas.height;
        let maxX = 0;
        let maxY = 0;
        console.log('Original image size:', {width: canvas.width, height: canvas.height})

        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const index = (y * canvas.width + x) * 4;
                const alpha = pixels[index + 3];
                if (alpha < 255) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        console.log('Image name:', img.src)
        console.log('Cropping coordinates:', {minX, minY, maxX, maxY });

        const croppedWidth = maxX - minX + 1;
        const croppedHeight = maxY - minY + 1;

        if (croppedWidth <= 0 || croppedHeight <= 0) {
            console.warn('No non-transparent pixels found in image:', img.src);
            return;
        }

        const croppedImageData = ctx.getImageData(minX, minY, croppedWidth, croppedHeight);

        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = croppedWidth;
        croppedCanvas.height = croppedHeight;
        const croppedCtx = croppedCanvas.getContext('2d');
        croppedCtx.putImageData(croppedImageData, 0, 0);

        img.src = croppedCanvas.toDataURL();
    });
};
