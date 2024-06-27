document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('blink-button');
    const body = document.body;
    const alarmSound = new Audio('static/alarm2.mp3'); // Adjust path as per your file structure

    let isBlinking = false;
    let isSoundPlaying = false;

    button.addEventListener('click', () => {
        isBlinking = !isBlinking;

        if (isBlinking) {
            body.classList.add('blinking-border');
            if (!isSoundPlaying) {
                alarmSound.loop = true;
                alarmSound.play();
                isSoundPlaying = true;
            }
        } else {
            body.classList.remove('blinking-border');
            alarmSound.pause();
            alarmSound.currentTime = 0; // Reset sound to start for next play
            isSoundPlaying = false;
        }
    });
});