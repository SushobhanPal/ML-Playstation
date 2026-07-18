document.addEventListener("DOMContentLoaded", () => {
    const menuToggle = document.getElementById("menuToggle");
    const menuClose = document.getElementById("menuClose");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("overlay");

    // Toggle menu
    function toggleMenu() {
        sidebar.classList.toggle("active");
        overlay.classList.toggle("active");
        menuToggle.classList.toggle("active");
    }

    // Close menu
    function closeMenu() {
        sidebar.classList.remove("active");
        overlay.classList.remove("active");
        menuToggle.classList.remove("active");
    }

    menuToggle.addEventListener("click", toggleMenu);
    menuClose.addEventListener("click", closeMenu);
    overlay.addEventListener("click", closeMenu);
});
