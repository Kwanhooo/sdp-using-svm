public class HelloWorld {
 /**
     * 增加配置
     */

    @PostMapping("/add")

    public void addConfig(HttpServletRequest request, HttpServletResponse response)

            throws IOException {

        response.setContentType("application/json;charset=UTF-8");

//返回了一个json格式的字符串。。

        Staaring result = configService.add(request);

        response.getOutputStream().write(result.getBytes());

        response.getOutputStream().flush();

    }


    /**
     * ！！！错误的示例
     *
     * @param request
     * @return
     */

    public String add(HttpServletRequest request) {

        Map data = new HashMap();

        try {

            String name = (String) request.getParameter("name");

            String value = (String) request.getParameter("value");

//示例代码

            long newID = add(name, value);

            data.put("code", 0);

            data.put("newID", newID);

        } catch (CheckException e) {

// 参数等校验出错，已知异常，不需要打印堆栈，返回码为-1

            data.put("code", -1);

            data.put("msg", e.getMessage());

        } catch (Exception e) {

// 其他未知异常，需要打印堆栈分析用，返回码为99

            logger.error("add config error", e);

            data.put("code", 99);

            data.put("msg", e.toString());

        }

        return JSONObject.toJSONString(data);

    }
}
