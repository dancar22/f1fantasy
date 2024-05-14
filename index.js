const buttons = document.querySelectorAll("a.driver");
buttons.forEach((button) => {
  button.addEventListener("click", () => {
    localStorage.setItem('driver-team', button.id);
  });
});