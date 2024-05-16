const buttons = document.querySelectorAll("a.driver");
buttons.forEach((button) => {
  button.addEventListener("click", () => {
    localStorage.setItem('team', button.id);
    localStorage.setItem('driver', 0);
  });
});