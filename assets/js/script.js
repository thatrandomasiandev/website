// Utility: Safe querySelector that returns null for invalid/empty selectors
function safeQuerySelector(selector) {
    if (typeof selector === 'string' && selector.trim() && selector !== '#') {
        try {
            return document.querySelector(selector);
        } catch (e) {
            return null;
        }
    }
    return null;
}

// Mobile Navigation Toggle
const hamburger = safeQuerySelector('.hamburger');
const navMenu = safeQuerySelector('.nav-menu');

if (hamburger && navMenu) {
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });
}

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
    const hamburger = safeQuerySelector('.hamburger');
    const navMenu = safeQuerySelector('.nav-menu');
    if (hamburger && navMenu) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    }
}));

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar background change on scroll - DISABLED to keep black background
// window.addEventListener('scroll', () => {
//     const navbar = safeQuerySelector('.navbar');
//     if (navbar && window.scrollY > 100) {
//         navbar.style.background = 'rgba(255, 255, 255, 0.98)';
//         navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
//     } else if (navbar) {
//         navbar.style.background = 'rgba(255, 255, 255, 0.95)';
//         navbar.style.boxShadow = 'none';
//     }
// });

// Portfolio Filtering
const filterButtons = document.querySelectorAll('.filter-btn');
const portfolioItems = document.querySelectorAll('.portfolio-item');

filterButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons
        filterButtons.forEach(btn => btn.classList.remove('active'));
        // Add active class to clicked button
        button.classList.add('active');
        
        const filter = button.getAttribute('data-filter');
        
        portfolioItems.forEach(item => {
            if (filter === 'all' || item.getAttribute('data-category') === filter) {
                item.style.display = 'block';
                item.style.animation = 'fadeIn 0.5s ease';
            } else {
                item.style.display = 'none';
            }
        });
    });
});

// Testimonials Slider
let currentSlide = 0;
const testimonialItems = document.querySelectorAll('.testimonial-item');
const navDots = document.querySelectorAll('.nav-dot');

function showSlide(index) {
    testimonialItems.forEach(item => {
        if (item && item.classList) {
            item.classList.remove('active');
        }
    });
    navDots.forEach(dot => {
        if (dot && dot.classList) {
            dot.classList.remove('active');
        }
    });
    
    if (testimonialItems[index]) {
        testimonialItems[index].classList.add('active');
    }
    if (navDots[index]) {
        navDots[index].classList.add('active');
    }
}

navDots.forEach((dot, index) => {
    if (dot) {
        dot.addEventListener('click', () => {
            currentSlide = index;
            showSlide(currentSlide);
        });
    }
});

// Auto-advance testimonials
setInterval(() => {
    currentSlide = (currentSlide + 1) % testimonialItems.length;
    showSlide(currentSlide);
}, 5000);

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.service-card, .stat, .contact-item, .portfolio-item, .team-member, .pricing-card, .blog-card');
    
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Form submission handling
const contactForm = safeQuerySelector('.contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(contactForm);
        const name = contactForm.querySelector('input[type="text"]').value;
        const email = contactForm.querySelector('input[type="email"]').value;
        const message = contactForm.querySelector('textarea').value;
        
        // Simple validation
        if (!name || !email || !message) {
            showNotification('Please fill in all fields', 'error');
            return;
        }
        
        if (!isValidEmail(email)) {
            showNotification('Please enter a valid email address', 'error');
            return;
        }
        
        // Simulate form submission
        showNotification('Message sent successfully!', 'success');
        contactForm.reset();
    });
}

// Email validation function
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Notification system
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = safeQuerySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Close button functionality
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    });
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Button click animations
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function(e) {
        // Create ripple effect
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        `;
        
        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Parallax effect for hero section - DISABLED to fix scrolling issues
// window.addEventListener('scroll', () => {
//     const scrolled = window.pageYOffset;
//     const hero = safeQuerySelector('.hero');
//     if (hero) {
//         const rate = scrolled * -0.5;
//         hero.style.transform = `translateY(${rate}px)`;
//     }
// });

// Counter animation for stats
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    function updateCounter() {
        start += increment;
        if (start < target) {
            element.textContent = Math.floor(start) + '+';
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target + '+';
        }
    }
    
    updateCounter();
}

// Trigger counter animation when stats section is visible
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const stats = entry.target.querySelectorAll('.stat h4');
            stats.forEach(stat => {
                const target = parseInt(stat.textContent);
                animateCounter(stat, target);
            });
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const statsSection = safeQuerySelector('.stats');
if (statsSection) {
    statsObserver.observe(statsSection);
}

// Add loading animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// Service card hover effects
document.querySelectorAll('.service-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Portfolio item hover effects
document.querySelectorAll('.portfolio-item').forEach(item => {
    item.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px)';
    });
    
    item.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Team member hover effects
document.querySelectorAll('.team-member').forEach(member => {
    member.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px)';
    });
    
    member.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Pricing card hover effects
document.querySelectorAll('.pricing-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        if (!this.classList.contains('featured')) {
            this.style.transform = 'translateY(-10px)';
        }
    });
    
    card.addEventListener('mouseleave', function() {
        if (!this.classList.contains('featured')) {
            this.style.transform = 'translateY(0)';
        }
    });
});

// Blog card hover effects
document.querySelectorAll('.blog-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Smooth reveal animations for sections
const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
    sectionObserver.observe(section);
});

// Add some interactive elements
console.log('Enhanced website loaded successfully! 🚀'); 

// Animated macOS-style moving squares background
(function() {
    const canvas = safeQuerySelector('#mac-bg-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = window.innerWidth;
    let height = window.innerHeight;
    let dpr = window.devicePixelRatio || 1;
    let squares = [];
    const NUM_SQUARES = 32;
    const SQUARE_MIN = 32;
    const SQUARE_MAX = 96;
    const LAYERS = 3;
    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }
    function randomSquare(layer) {
        const size = SQUARE_MIN + Math.random() * (SQUARE_MAX - SQUARE_MIN) * (1 - layer * 0.2);
        const solid = Math.random() < 0.5;
        return {
            x: Math.random() * width,
            y: Math.random() * height,
            size,
            speed: 0.15 + Math.random() * 0.12 + layer * 0.07,
            angle: Math.random() * Math.PI * 2,
            drift: 0.5 + Math.random() * 0.7,
            solid,
            opacity: 0.12 + Math.random() * 0.18 + layer * 0.08,
            layer
        };
    }
    function initSquares() {
        squares = [];
        for (let l = 0; l < LAYERS; l++) {
            for (let i = 0; i < NUM_SQUARES / LAYERS; i++) {
                squares.push(randomSquare(l));
            }
        }
    }
    function draw() {
        ctx.clearRect(0, 0, width, height);
        const t = Date.now() * 0.00025;
        for (const sq of squares) {
            // Animate position
            const px = sq.x + Math.sin(t * sq.speed + sq.angle) * 40 * sq.drift * (1 + sq.layer * 0.3);
            const py = sq.y + Math.cos(t * sq.speed + sq.angle) * 40 * sq.drift * (1 + sq.layer * 0.3);
            ctx.save();
            ctx.globalAlpha = sq.opacity;
            ctx.lineWidth = 2 + sq.layer;
            if (sq.solid) {
                ctx.fillStyle = '#111';
                ctx.fillRect(px, py, sq.size, sq.size);
            } else {
                ctx.strokeStyle = '#111';
                ctx.strokeRect(px, py, sq.size, sq.size);
            }
            ctx.restore();
        }
        requestAnimationFrame(draw);
    }
    window.addEventListener('resize', () => {
        resize();
        initSquares();
    });
    resize();
    initSquares();
    draw();
})(); 

// About section typing animation
(function() {
    const code = `class engineer:
    def __init__(my):
        my.name = "Joshua"
        my.position = "Computer Science Student @ USC"
        my.hobbies = [
            "developing and manufacturing ideas into reality",
            "travelling",
            "decorating my Hydro Flask",
            "hanging out with friends",
    
            "3D printing",
            "Playing Tennis"
        ]
        
        my.skills_header = "SKILLS & TECHNOLOGIES"
        my.skills = [
            "Computer-Aided Design/3D Printing",
            "Python/C++",
            "Swift/SwiftUI",
            "HTML/CSS",
            "JavaScript",
        ]
        my.operating_systems = [
            "macOS",
            "Windows"
        ]`;
    const block = safeQuerySelector('#about-code-block');
    if (!block) return;
    let idx = 0;
    let cursor = true;
    function type() {
        block.innerHTML =
            '<pre style="margin:0;background:#0e0e0e;color:#ff6b35;border-radius:18px;padding:2rem;font-size:1.05rem;box-shadow:none;height:100%;min-height:400px;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-start;">' +
            code.slice(0, idx) +
            (cursor ? '<span style="color:#ff6b35;">|</span>' : '<span style="color:#ff6b35;opacity:0.2;">|</span>') +
            '</pre>';
        if (idx < code.length) {
            idx++;
            setTimeout(type, 18 + Math.random()*40);
        } else {
            setTimeout(() => { cursor = !cursor; type(); }, 500);
        }
    }
    type();
})(); 

// === PDF Dropdown and Direct Link Logic (No Embedded Viewer) ===
document.addEventListener('DOMContentLoaded', () => {
    const pdfs = [
        { name: 'Discrete Math Course Project (Tower of Hanoi)', file: 'Discrete_Math_Course_Project_Tower_of_Hanoi_V2.pdf' },
        { name: 'Radiation', file: 'Radiation.pdf' },
        { name: 'Cold Welding', file: 'Cold_Welding.pdf' }
    ];
    const menu = document.getElementById('pdf-menu');
    const viewerLink = document.getElementById('pdfjs-viewer-link');
    const viewer = document.getElementById('pdfjs-viewer');
    if (!menu || !viewerLink || !viewer) return;
    // Populate dropdown
    menu.innerHTML = '<option value="">Select a PDF...</option>' +
        pdfs.map(pdf => `<option value="${pdf.file}">${pdf.name}</option>`).join('');
    // On change, update direct link
    menu.addEventListener('change', function() {
        const file = this.value;
        if (file) {
            viewerLink.href = `pdfs/${encodeURIComponent(file)}`;
            viewerLink.style.display = 'block';
            viewerLink.textContent = 'Open PDF';
            viewerLink.removeAttribute('target');
            viewerLink.removeAttribute('rel');
            viewer.innerHTML = '';
        } else {
            viewerLink.href = '#';
            viewerLink.style.display = 'none';
            viewer.innerHTML = '';
        }
    });
    // Hide link if no PDF selected
    viewerLink.style.display = 'none';
    // Hide embedded viewer area
    viewer.style.display = 'none';
}); 

// Counter animation for hero stats
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    function updateCounter() {
        start += increment;
        if (start < target) {
            element.textContent = Math.floor(start) + '+';
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target + '+';
        }
    }
    
    updateCounter();
}

// Trigger counter animation when hero section is visible
const heroStatsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const stats = entry.target.querySelectorAll('.hero-stat-number');
            stats.forEach(stat => {
                const target = parseInt(stat.textContent);
                animateCounter(stat, target);
            });
            heroStatsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

// Observe hero section for stats animation
const heroSection = document.querySelector('.hero');
if (heroSection) {
    heroStatsObserver.observe(heroSection);
}

// Skill stats counter animation
const skillStatsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const skillStats = entry.target.querySelectorAll('.skill-stat-number');
            skillStats.forEach(stat => {
                const target = parseInt(stat.getAttribute('data-target'));
                animateCounter(stat, target);
            });
            skillStatsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

// Observe about section for skill stats animation
const aboutSection = document.querySelector('.about');
if (aboutSection) {
    skillStatsObserver.observe(aboutSection);
}

// Observe skills section for skill stats animation
const skillsSection = document.querySelector('.services');
if (skillsSection) {
    skillStatsObserver.observe(skillsSection);
}

// Blog Category Filtering
document.addEventListener('DOMContentLoaded', () => {
    const categoryCards = document.querySelectorAll('.category-card');
    const blogCards = document.querySelectorAll('.blog-card');
    
    if (categoryCards.length === 0 || blogCards.length === 0) return;
    
    categoryCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove active class from all category cards
            categoryCards.forEach(c => c.classList.remove('active'));
            // Add active class to clicked card
            card.classList.add('active');
            
            const category = card.querySelector('span').textContent.toLowerCase();
            
            // Filter blog cards based on category
            blogCards.forEach(blogCard => {
                const blogCategory = blogCard.querySelector('.blog-category').textContent.toLowerCase();
                
                if (category === 'all' || blogCategory.includes(category) || 
                    (category === 'ai & ml' && (blogCategory.includes('ai') || blogCategory.includes('ml'))) ||
                    (category === 'computer science' && (blogCategory.includes('computer') || blogCategory.includes('science'))) ||
                    (category === 'mechanical engineering' && (blogCategory.includes('mechanical') || blogCategory.includes('engineering'))) ||
                    (category === 'robotics' && blogCategory.includes('robotics')) ||
                    (category === 'spacecraft' && blogCategory.includes('spacecraft'))) {
                    
                    blogCard.style.display = 'block';
                    blogCard.style.animation = 'fadeIn 0.5s ease';
                } else {
                    blogCard.style.display = 'none';
                }
            });
        });
    });
    
    // Add "All" category option
    const categoriesGrid = document.querySelector('.categories-grid');
    if (categoriesGrid) {
        const allCategory = document.createElement('div');
        allCategory.className = 'category-card active';
        allCategory.innerHTML = '<i class="fas fa-th-large"></i><span>All</span>';
        allCategory.addEventListener('click', () => {
            categoryCards.forEach(c => c.classList.remove('active'));
            allCategory.classList.add('active');
            
            blogCards.forEach(blogCard => {
                blogCard.style.display = 'block';
                blogCard.style.animation = 'fadeIn 0.5s ease';
            });
        });
        categoriesGrid.insertBefore(allCategory, categoriesGrid.firstChild);
    }
});

// Newsletter form submission
document.addEventListener('DOMContentLoaded', () => {
    const newsletterForm = document.querySelector('.newsletter-form');
    if (newsletterForm) {
        newsletterForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const email = newsletterForm.querySelector('input[type="email"]').value;
            
            // Simple validation
            if (email && email.includes('@')) {
                // Show success message (you can customize this)
                alert('Thank you for subscribing to our newsletter!');
                newsletterForm.reset();
            } else {
                alert('Please enter a valid email address.');
            }
        });
    }
});

// Count-up animation for skills stats
function animateCountUp() {
    const counters = document.querySelectorAll('.skill-stat h4[data-target]');
    
    if (counters.length === 0) return;
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 1000; // 1 second
        const step = target / (duration / 16); // 60fps
        let current = 0;
        
        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            counter.textContent = Math.floor(current);
        }, 16);
    });
}

// Start count-up when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Start count-up immediately for skills page
    animateCountUp();
    
    // Also observe for skills section if it's not immediately visible
    const skillsSection = document.querySelector('#services');
    if (skillsSection) {
        const skillsObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCountUp();
                    skillsObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });
        
        skillsObserver.observe(skillsSection);
    }
});


// Dropdown Menu Functionality
document.addEventListener('DOMContentLoaded', () => {
    // Navigation dropdown functionality
    const navDropdowns = document.querySelectorAll('.nav-item.dropdown');
    
    navDropdowns.forEach(dropdown => {
        const toggle = dropdown.querySelector('.dropdown-toggle');
        const menu = dropdown.querySelector('.dropdown-menu');
        
        if (toggle && menu) {
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // Close other dropdowns
                navDropdowns.forEach(other => {
                    if (other !== dropdown) {
                        other.querySelector('.dropdown-menu').classList.remove('show');
                        other.querySelector('.dropdown-toggle').setAttribute('aria-expanded', 'false');
                    }
                });
                
                // Toggle current dropdown
                const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
                toggle.setAttribute('aria-expanded', !isExpanded);
                menu.classList.toggle('show');
            });
        }
    });
    
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.dropdown')) {
            // Close all navigation dropdowns
            navDropdowns.forEach(dropdown => {
                const menu = dropdown.querySelector('.dropdown-menu');
                const toggle = dropdown.querySelector('.dropdown-toggle');
                if (menu && toggle) {
                    menu.classList.remove('show');
                    toggle.setAttribute('aria-expanded', 'false');
                }
            });
            
        }
    });
    
    // Close dropdowns on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            navDropdowns.forEach(dropdown => {
                const menu = dropdown.querySelector('.dropdown-menu');
                const toggle = dropdown.querySelector('.dropdown-toggle');
                if (menu && toggle) {
                    menu.classList.remove('show');
                    toggle.setAttribute('aria-expanded', 'false');
                }
            });
            
        }
    });
});



