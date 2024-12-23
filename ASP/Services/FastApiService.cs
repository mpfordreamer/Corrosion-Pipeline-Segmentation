using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace ASP.Services
{
    public class FastApiService
    {
        private readonly HttpClient _httpClient;

        public FastApiService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<string> SendImageForSegmentation(IFormFile image, string segmentationType)
        {
            var formData = new MultipartFormDataContent();
            var fileContent = new ByteArrayContent(await GetImageBytes(image));
            fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg"); // Change to "image/jpeg" if your API expects JPEG

            formData.Add(fileContent, "file", image.FileName);

            // Construct the URL based on the segmentation type
            var url = $"http://localhost:8000/segment/{segmentationType}/";
            var response = await _httpClient.PostAsync(url, formData);

            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadAsStringAsync();
                return result;  
            }

            return null; // Return null if the request fails
        }

        private async Task<byte[]> GetImageBytes(IFormFile image)
        {
            using (var memoryStream = new System.IO.MemoryStream())
            {
                await image.CopyToAsync(memoryStream);
                return memoryStream.ToArray();
            }
        }
    }
}