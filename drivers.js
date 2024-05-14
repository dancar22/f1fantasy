const buttons = document.querySelectorAll("a.driver");
buttons.forEach((button) => {
  button.addEventListener("click", () => {
    localStorage.setItem('driver', button.id);
  });
});