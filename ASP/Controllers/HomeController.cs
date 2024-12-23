using Microsoft.AspNetCore.Mvc;
using UjiCoba.Models;
using System.Net.Http;
using System.Net.Http.Headers;
using Newtonsoft.Json;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Extensions.Configuration;
using Microsoft.AspNetCore.Hosting;
using System.Text.Json;
using System;

namespace ASP.Controllers
{
    public class HomeController : Controller
    {
        private readonly HttpClient _httpClient;
        private readonly string _fastApiUrl;
        private readonly IWebHostEnvironment _hostingEnvironment;

        public HomeController(HttpClient httpClient, IConfiguration configuration, IWebHostEnvironment hostingEnvironment)
        {
            _httpClient = httpClient;
            _fastApiUrl = configuration.GetValue<string>("FastApiUrl") 
                          ?? throw new ArgumentNullException(nameof(_fastApiUrl), "FastApiUrl must be configured.");
            _hostingEnvironment = hostingEnvironment;
        }

        public IActionResult Index() => View();

        public IActionResult About() => View();

        public IActionResult ListImage() => View();

        [HttpPost]
        public async Task<IActionResult> UploadImages(UploadImage model)
        {
            if (model.ImageFiles == null || model.ImageFiles.Count == 0)
            {
                ModelState.AddModelError("", "No files selected for upload.");
                return View("Index");
            }

            var results = new List<string>();

            foreach (var imageFile in model.ImageFiles)
            {
                try
                {
                    // Upload image to FastAPI
                    var uploadResult = await SendImageToFastApi(imageFile);
                    if (uploadResult == null)
                    {
                        ModelState.AddModelError("", $"Failed to upload image: {imageFile.FileName}");
                        continue;
                    }

                    // Call segmentation endpoint
                    var segmentationResult = await GetSegmentationResult(model.SegmentationType);
                    if (segmentationResult != null)
                    {
                        results.Add(segmentationResult);
                    }
                }
                catch (Exception ex)
                {
                    ModelState.AddModelError("", $"Error processing image {imageFile.FileName}: {ex.Message}");
                }
            }

            ViewBag.SegmentationResults = results;
            ViewBag.SelectedSegmentationType = model.SegmentationType;

            return View("UploadResult");
        }

        private async Task<string?> SendImageToFastApi(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
                return null;

            var formContent = new MultipartFormDataContent();

            using (var ms = new MemoryStream())
            {
                await imageFile.CopyToAsync(ms);
                var byteArray = ms.ToArray();

                var byteContent = new ByteArrayContent(byteArray)
                {
                    Headers = { ContentType = MediaTypeHeaderValue.Parse("image/jpeg") }
                };

                formContent.Add(byteContent, "file", imageFile.FileName);

                var response = await _httpClient.PostAsync($"{_fastApiUrl}/upload/", formContent);

                return response.IsSuccessStatusCode ? await response.Content.ReadAsStringAsync() : null;
            }
        }

        // Mengambil hasil segmentasi berdasarkan jenis segmentasi
        private async Task<string?> GetSegmentationResult(string segmentationType)
        {
            if (string.IsNullOrEmpty(segmentationType))
                return null;

            try
            {
                var segmentationUrl = $"{_fastApiUrl}/segment/{segmentationType}/";
                var response = await _httpClient.GetAsync(segmentationUrl);

                if (!response.IsSuccessStatusCode)
                {
                    throw new HttpRequestException($"Failed to fetch segmentation result for type: {segmentationType}");
                }

                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return $"data:image/jpeg;base64,{Convert.ToBase64String(imageBytes)}";
            }
            catch (Exception ex)
            {
                // Logging error if needed
                Console.WriteLine($"Error in GetSegmentationResult: {ex.Message}");
                return null;
            }
        }

        // Show data statistics after segmentation
            [HttpGet("HomeController/Segmentdata")]
            public async Task<IActionResult> SegmentData()
            {
                var response = await _httpClient.GetAsync($"{_fastApiUrl}/segment/data/");
                if (!response.IsSuccessStatusCode)
                {
                    return StatusCode(500, "Error while fetching segment data.");
                }
                var result = await response.Content.ReadAsStringAsync();
                var dataData = JsonConvert.DeserializeObject<Dictionary<string, string>>(result);
                return Json(dataData);
            }


        [HttpPost]
        public async Task<IActionResult> DownloadImages([FromBody] DownloadRequest request)
        {
            if (request == null || string.IsNullOrEmpty(request.PredictionType) || request.Files == null || request.Files.Count == 0)
            {
                return BadRequest(new { message = "Invalid request." });
            }

            var downloadDirectory = Path.Combine(_hostingEnvironment.WebRootPath, "DownloadImages");
            Directory.CreateDirectory(downloadDirectory);

            var savedFiles = new List<string>();

            foreach (var imageUrl in request.Files)
            {
                if (string.IsNullOrWhiteSpace(imageUrl) || imageUrl.StartsWith("data:image/jpeg;base64,"))
                {
                    continue;
                }

                var fileName = Path.GetFileName(imageUrl);
                var fileExtension = Path.GetExtension(fileName).ToLower();

                if (fileExtension != ".png" && fileExtension != ".jpg" && fileExtension != ".jpeg")
                {
                    fileName = Path.ChangeExtension(fileName, ".jpg");
                }

                var filePath = Path.Combine(downloadDirectory, fileName);

                try
                {
                    var imageBytes = await _httpClient.GetByteArrayAsync(imageUrl);
                    if (imageBytes == null || imageBytes.Length == 0)
                    {
                        Console.WriteLine($"No data returned for URL: {imageUrl}");
                        continue;
                    }

                    await System.IO.File.WriteAllBytesAsync(filePath, imageBytes);
                    savedFiles.Add(filePath);
                    Console.WriteLine($"Saved file: {filePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error downloading image from {imageUrl}: {ex.Message}");
                    return StatusCode(500, new { message = "Error downloading images." });
                }
            }

            return Ok(new { files = savedFiles });
        }
    }
}
