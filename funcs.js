const btn = document.querySelector("#btn");
const btn2 = document.querySelector("#btn2");

let driver = "none";
let track = "none";


btn.addEventListener("click", () => {
    driver = 0;
    localStorage.setItem('driver', driver);
});

btn2.addEventListener("click", () => {
    track = 0;
    localStorage.setItem('track', track);
    alert("Driver: " + localStorage.getItem('driver') + "\n" + "Track: " + localStorage.getItem('track') + "\nTeam/Driver: " + localStorage.getItem('driver-team'));

});


