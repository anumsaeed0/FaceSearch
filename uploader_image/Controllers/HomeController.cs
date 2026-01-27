using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Hosting;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using System.Text.Json;
using System.Diagnostics;

namespace uploader_image.Controllers
{
    public class HomeController : Controller
    {
        private readonly IWebHostEnvironment _env;

        public HomeController(IWebHostEnvironment env)
        {
            _env = env;
        }

        
        // GET: Upload Page
        [HttpGet]
        public IActionResult Index()
        {
            var uploadsFolder = Path.Combine(_env.WebRootPath, "uploads");

            if (Directory.Exists(uploadsFolder))
            {
                var file = Directory.GetFiles(uploadsFolder).FirstOrDefault();
                if (file != null)
                {
                    ViewBag.ImagePath = "/uploads/" + Path.GetFileName(file);
                    ViewBag.FileName = Path.GetFileName(file);
                }
            }

            return View();
        }

        
        // POST: Upload Image (MVC)
        [HttpPost]
        public async Task<IActionResult> Index(Microsoft.AspNetCore.Http.IFormFile imageFile)
        {
            var uploadsFolder = Path.Combine(_env.WebRootPath, "uploads");
            if (!Directory.Exists(uploadsFolder))
                Directory.CreateDirectory(uploadsFolder);

            foreach (var file in Directory.GetFiles(uploadsFolder))
                System.IO.File.Delete(file);

            if (imageFile == null || imageFile.Length == 0)
            {
                ViewBag.Message = "Please select an image.";
                return View();
            }

            var filePath = Path.Combine(uploadsFolder, imageFile.FileName);

            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await imageFile.CopyToAsync(stream);
            }

            ViewBag.Message = "Image uploaded successfully!";
            ViewBag.ImagePath = "/uploads/" + imageFile.FileName;
            ViewBag.FileName = imageFile.FileName;

            return View();
        }

        
        // GET: List Images for Python monitor
        [HttpGet("api/list-images")]
        public IActionResult ListImages()
        {
            var uploadsFolder = Path.Combine(_env.WebRootPath, "uploads");
            if (!Directory.Exists(uploadsFolder))
                return Ok(new string[] { });

            var urls = Directory.GetFiles(uploadsFolder)
                .Select(f => $"{Request.Scheme}://{Request.Host}/uploads/{Path.GetFileName(f)}")
                .ToArray();

            //Process p = new Process();
            //p.StartInfo = new ProcessStartInfo();
            //p.StartInfo.ArgumentList

            return Ok(urls);
        }

        
        // GET: Face Match Results (JSON)
        [HttpGet("api/face-match-results")]
        public IActionResult GetFaceMatchResults()
        {
            var resultsFolder = Path.Combine(_env.WebRootPath, "results");
            var resultsFile = Path.Combine(resultsFolder, "latest_match.json");

            if (!System.IO.File.Exists(resultsFile))
                return Ok(new { success = false, message = "No results available." });

            try
            {
                var jsonContent = System.IO.File.ReadAllText(resultsFile);
                var result = JsonSerializer.Deserialize<object>(jsonContent);
                return Ok(result);
            }
            catch
            {
                return Ok(new { success = false, message = "Failed to read results." });
            }
        }

        
        // POST: Delete Image
        [HttpPost]
        public IActionResult DeleteImage(string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
                return RedirectToAction("Index");

            var uploadsFolder = Path.Combine(_env.WebRootPath, "uploads");
            var filePath = Path.Combine(uploadsFolder, fileName);

            if (System.IO.File.Exists(filePath))
                System.IO.File.Delete(filePath);

            return RedirectToAction("Index");
        }
    }
}
