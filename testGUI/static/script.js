document.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('keydown', (event) => {
        if (event.key === 'p') {
            const videoContainer = document.querySelector('.video-container');
            videoContainer.classList.toggle('flashing');
        }
    });
});